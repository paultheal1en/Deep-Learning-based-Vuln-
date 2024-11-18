int kvm_irqchip_remove_irq_notifier(KVMState *s, EventNotifier *n, int virq)

{

    return kvm_irqchip_remove_irqfd(s, event_notifier_get_fd(n), virq);

}
