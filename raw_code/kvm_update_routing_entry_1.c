static int kvm_update_routing_entry(KVMState *s,

                                    struct kvm_irq_routing_entry *new_entry)

{

    struct kvm_irq_routing_entry *entry;

    int n;



    for (n = 0; n < s->irq_routes->nr; n++) {

        entry = &s->irq_routes->entries[n];

        if (entry->gsi != new_entry->gsi) {

            continue;

        }



        entry->type = new_entry->type;

        entry->flags = new_entry->flags;

        entry->u = new_entry->u;



        kvm_irqchip_commit_routes(s);



        return 0;

    }



    return -ESRCH;

}
