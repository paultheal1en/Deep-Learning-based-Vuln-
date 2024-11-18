static void apic_update_irq(APICCommonState *s)

{

    if (!(s->spurious_vec & APIC_SV_ENABLE)) {

        return;

    }

    if (apic_irq_pending(s) > 0) {

        cpu_interrupt(s->cpu_env, CPU_INTERRUPT_HARD);

    } else if (apic_accept_pic_intr(&s->busdev.qdev) &&

               pic_get_output(isa_pic)) {

        apic_deliver_pic_intr(&s->busdev.qdev, 1);

    }

}
