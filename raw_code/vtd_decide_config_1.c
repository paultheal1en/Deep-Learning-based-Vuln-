static bool vtd_decide_config(IntelIOMMUState *s, Error **errp)

{

    X86IOMMUState *x86_iommu = X86_IOMMU_DEVICE(s);



    /* Currently Intel IOMMU IR only support "kernel-irqchip={off|split}" */

    if (x86_iommu->intr_supported && kvm_irqchip_in_kernel() &&

        !kvm_irqchip_is_split()) {

        error_setg(errp, "Intel Interrupt Remapping cannot work with "

                         "kernel-irqchip=on, please use 'split|off'.");

        return false;

    }

    if (s->intr_eim == ON_OFF_AUTO_ON && !x86_iommu->intr_supported) {

        error_setg(errp, "eim=on cannot be selected without intremap=on");

        return false;

    }



    if (s->intr_eim == ON_OFF_AUTO_AUTO) {

        s->intr_eim = x86_iommu->intr_supported ?

                                              ON_OFF_AUTO_ON : ON_OFF_AUTO_OFF;

    }



    return true;

}
