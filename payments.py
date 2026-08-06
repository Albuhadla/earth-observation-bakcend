"""
Earth Observation and Analysis — Payments (Stripe)
=========================================
"""
import os, stripe
from flask import Blueprint, request, jsonify, current_app
from models import db, User
from auth import token_required

payments_bp = Blueprint('payments', __name__)
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')

PRICE_IDS = {
    'basic':      os.getenv('STRIPE_PRICE_BASIC', ''),
    'pro':        os.getenv('STRIPE_PRICE_PRO', ''),
    'enterprise': os.getenv('STRIPE_PRICE_ENTERPRISE', ''),
}
WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET', '')


def create_stripe_customer(user):
    if not stripe.api_key: return None
    try:
        c = stripe.Customer.create(email=user.email, name=user.name,
                                    metadata={'user_id': str(user.id)})
        return c.id
    except stripe.error.StripeError as e:
        current_app.logger.error(f'Stripe customer error: {e}')
        return None


@payments_bp.route('/subscribe', methods=['POST'])
@token_required
def subscribe(user):
    d = request.get_json()
    plan = d.get('plan', 'basic')
    if plan not in PRICE_IDS:
        return jsonify({'error': 'Invalid plan.'}), 400

    if not stripe.api_key:
        # Demo mode — no real Stripe configured, just flip the plan
        user.plan = plan; user.plan_status = 'active'
        db.session.commit()
        return jsonify({'message': f'Plan set to {plan} (demo mode — no Stripe key configured).', 'user': user.to_dict()})

    try:
        if not user.stripe_customer_id:
            cid = create_stripe_customer(user)
            if cid: user.stripe_customer_id = cid; db.session.commit()

        pm_id = d.get('payment_method_id')
        if pm_id:
            stripe.PaymentMethod.attach(pm_id, customer=user.stripe_customer_id)
            stripe.Customer.modify(user.stripe_customer_id, invoice_settings={'default_payment_method': pm_id})

        if user.stripe_sub_id:
            try: stripe.Subscription.cancel(user.stripe_sub_id)
            except Exception: pass

        sub = stripe.Subscription.create(
            customer=user.stripe_customer_id,
            items=[{'price': PRICE_IDS[plan]}],
            trial_period_days=14,
            metadata={'user_id': str(user.id), 'plan': plan}
        )
        user.plan = plan; user.plan_status = 'trial'; user.stripe_sub_id = sub.id
        db.session.commit()
        return jsonify({'message': f'Subscribed to {plan} with 14-day trial.', 'user': user.to_dict()})
    except stripe.error.CardError as e:
        return jsonify({'error': f'Card error: {e.user_message}'}), 402
    except stripe.error.StripeError as e:
        return jsonify({'error': str(e)}), 500


@payments_bp.route('/webhook', methods=['POST'])
def webhook():
    payload, sig = request.get_data(), request.headers.get('Stripe-Signature', '')
    try:
        event = (stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET) if WEBHOOK_SECRET
                 else stripe.Event.construct_from(request.get_json(), stripe.api_key))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    obj, etype = event['data']['object'], event['type']

    if etype in ('customer.subscription.updated', 'invoice.payment_succeeded'):
        user = User.query.filter_by(stripe_customer_id=obj['customer']).first()
        if user:
            user.plan_status = 'active'
            db.session.commit()
    elif etype == 'customer.subscription.deleted':
        user = User.query.filter_by(stripe_customer_id=obj['customer']).first()
        if user:
            user.plan_status = 'cancelled'; user.stripe_sub_id = None
            db.session.commit()
    elif etype == 'invoice.payment_failed':
        user = User.query.filter_by(stripe_customer_id=obj['customer']).first()
        if user:
            user.plan_status = 'past_due'
            db.session.commit()

    return jsonify({'received': True})
