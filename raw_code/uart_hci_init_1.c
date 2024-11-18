CharDriverState *uart_hci_init(qemu_irq wakeup)

{

    struct csrhci_s *s = (struct csrhci_s *)

            g_malloc0(sizeof(struct csrhci_s));



    s->chr.opaque = s;

    s->chr.chr_write = csrhci_write;

    s->chr.chr_ioctl = csrhci_ioctl;




    s->hci = qemu_next_hci();

    s->hci->opaque = s;

    s->hci->evt_recv = csrhci_out_hci_packet_event;

    s->hci->acl_recv = csrhci_out_hci_packet_acl;



    s->out_tm = qemu_new_timer_ns(vm_clock, csrhci_out_tick, s);

    s->pins = qemu_allocate_irqs(csrhci_pins, s, __csrhci_pins);

    csrhci_reset(s);



    return &s->chr;

}