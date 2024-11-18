vubr_set_vring_addr_exec(VubrDev *dev, VhostUserMsg *vmsg)
{
    struct vhost_vring_addr *vra = &vmsg->payload.addr;
    unsigned int index = vra->index;
    VubrVirtq *vq = &dev->vq[index];
    DPRINT("vhost_vring_addr:\n");
    DPRINT("    index:  %d\n", vra->index);
    DPRINT("    flags:  %d\n", vra->flags);
    DPRINT("    desc_user_addr:   0x%016llx\n", vra->desc_user_addr);
    DPRINT("    used_user_addr:   0x%016llx\n", vra->used_user_addr);
    DPRINT("    avail_user_addr:  0x%016llx\n", vra->avail_user_addr);
    DPRINT("    log_guest_addr:   0x%016llx\n", vra->log_guest_addr);
    vq->desc = (struct vring_desc *)(uintptr_t)qva_to_va(dev, vra->desc_user_addr);
    vq->used = (struct vring_used *)(uintptr_t)qva_to_va(dev, vra->used_user_addr);
    vq->avail = (struct vring_avail *)(uintptr_t)qva_to_va(dev, vra->avail_user_addr);
    vq->log_guest_addr = vra->log_guest_addr;
    DPRINT("Setting virtq addresses:\n");
    DPRINT("    vring_desc  at %p\n", vq->desc);
    DPRINT("    vring_used  at %p\n", vq->used);
    DPRINT("    vring_avail at %p\n", vq->avail);
    vq->last_used_index = vq->used->idx;
    return 0;