static void machine_set_kernel_irqchip(Object *obj, bool value, Error **errp)

{

    MachineState *ms = MACHINE(obj);



    ms->kernel_irqchip = value;

}
