MemTxAttrs kvm_arch_post_run(CPUState *cs, struct kvm_run *run)

{

    ARMCPU *cpu;

    uint32_t switched_level;



    if (kvm_irqchip_in_kernel()) {

        /*

         * We only need to sync timer states with user-space interrupt

         * controllers, so return early and save cycles if we don't.

         */

        return MEMTXATTRS_UNSPECIFIED;

    }



    cpu = ARM_CPU(cs);



    /* Synchronize our shadowed in-kernel device irq lines with the kvm ones */

    if (run->s.regs.device_irq_level != cpu->device_irq_level) {

        switched_level = cpu->device_irq_level ^ run->s.regs.device_irq_level;



        qemu_mutex_lock_iothread();



        if (switched_level & KVM_ARM_DEV_EL1_VTIMER) {

            qemu_set_irq(cpu->gt_timer_outputs[GTIMER_VIRT],

                         !!(run->s.regs.device_irq_level &

                            KVM_ARM_DEV_EL1_VTIMER));

            switched_level &= ~KVM_ARM_DEV_EL1_VTIMER;

        }



        if (switched_level & KVM_ARM_DEV_EL1_PTIMER) {

            qemu_set_irq(cpu->gt_timer_outputs[GTIMER_PHYS],

                         !!(run->s.regs.device_irq_level &

                            KVM_ARM_DEV_EL1_PTIMER));

            switched_level &= ~KVM_ARM_DEV_EL1_PTIMER;

        }



        /* XXX PMU IRQ is missing */



        if (switched_level) {

            qemu_log_mask(LOG_UNIMP, "%s: unhandled in-kernel device IRQ %x\n",

                          __func__, switched_level);

        }



        /* We also mark unknown levels as processed to not waste cycles */

        cpu->device_irq_level = run->s.regs.device_irq_level;

        qemu_mutex_unlock_iothread();

    }



    return MEMTXATTRS_UNSPECIFIED;

}
