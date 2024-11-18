int kvm_arch_add_msi_route_post(struct kvm_irq_routing_entry *route,
                                int vector, PCIDevice *dev)
{
    static bool notify_list_inited = false;
    MSIRouteEntry *entry;
    if (!dev) {
        /* These are (possibly) IOAPIC routes only used for split
         * kernel irqchip mode, while what we are housekeeping are
         * PCI devices only. */
        return 0;
    entry = g_new0(MSIRouteEntry, 1);
    entry->dev = dev;
    entry->vector = vector;
    entry->virq = route->gsi;
    QLIST_INSERT_HEAD(&msi_route_list, entry, list);
    trace_kvm_x86_add_msi_route(route->gsi);
    return 0;