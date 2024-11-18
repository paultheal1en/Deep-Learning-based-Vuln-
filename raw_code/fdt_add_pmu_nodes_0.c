static void fdt_add_pmu_nodes(const VirtMachineState *vms)

{

    CPUState *cpu;

    ARMCPU *armcpu;

    uint32_t irqflags = GIC_FDT_IRQ_FLAGS_LEVEL_HI;



    CPU_FOREACH(cpu) {

        armcpu = ARM_CPU(cpu);

        if (!arm_feature(&armcpu->env, ARM_FEATURE_PMU) ||

            !kvm_arm_pmu_create(cpu, PPI(VIRTUAL_PMU_IRQ))) {

            return;

        }

    }



    if (vms->gic_version == 2) {

        irqflags = deposit32(irqflags, GIC_FDT_IRQ_PPI_CPU_START,

                             GIC_FDT_IRQ_PPI_CPU_WIDTH,

                             (1 << vms->smp_cpus) - 1);

    }



    armcpu = ARM_CPU(qemu_get_cpu(0));

    qemu_fdt_add_subnode(vms->fdt, "/pmu");

    if (arm_feature(&armcpu->env, ARM_FEATURE_V8)) {

        const char compat[] = "arm,armv8-pmuv3";

        qemu_fdt_setprop(vms->fdt, "/pmu", "compatible",

                         compat, sizeof(compat));

        qemu_fdt_setprop_cells(vms->fdt, "/pmu", "interrupts",

                               GIC_FDT_IRQ_TYPE_PPI, VIRTUAL_PMU_IRQ, irqflags);

    }

}
