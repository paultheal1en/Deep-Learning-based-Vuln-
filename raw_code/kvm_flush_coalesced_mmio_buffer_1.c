void kvm_flush_coalesced_mmio_buffer(void)

{

#ifdef KVM_CAP_COALESCED_MMIO

    KVMState *s = kvm_state;

    if (s->coalesced_mmio_ring) {

        struct kvm_coalesced_mmio_ring *ring = s->coalesced_mmio_ring;

        while (ring->first != ring->last) {

            struct kvm_coalesced_mmio *ent;



            ent = &ring->coalesced_mmio[ring->first];



            cpu_physical_memory_write(ent->phys_addr, ent->data, ent->len);

            smp_wmb();

            ring->first = (ring->first + 1) % KVM_COALESCED_MMIO_MAX;

        }

    }

#endif

}
