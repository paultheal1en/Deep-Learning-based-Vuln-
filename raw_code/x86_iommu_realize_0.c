static void x86_iommu_realize(DeviceState *dev, Error **errp)

{

    X86IOMMUState *x86_iommu = X86_IOMMU_DEVICE(dev);

    X86IOMMUClass *x86_class = X86_IOMMU_GET_CLASS(dev);

    MachineState *ms = MACHINE(qdev_get_machine());

    MachineClass *mc = MACHINE_GET_CLASS(ms);

    PCMachineState *pcms =

        PC_MACHINE(object_dynamic_cast(OBJECT(ms), TYPE_PC_MACHINE));

    QLIST_INIT(&x86_iommu->iec_notifiers);



    if (!pcms) {

        error_setg(errp, "Machine-type '%s' not supported by IOMMU",

                   mc->name);

        return;

    }



    if (x86_class->realize) {

        x86_class->realize(dev, errp);

    }



    x86_iommu_set_default(X86_IOMMU_DEVICE(dev));

}
