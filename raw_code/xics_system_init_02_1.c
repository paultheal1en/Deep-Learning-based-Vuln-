static XICSState *xics_system_init(int nr_servers, int nr_irqs)

{

    XICSState *icp = NULL;



    if (kvm_enabled()) {

        QemuOpts *machine_opts = qemu_get_machine_opts();

        bool irqchip_allowed = qemu_opt_get_bool(machine_opts,

                                                "kernel_irqchip", true);

        bool irqchip_required = qemu_opt_get_bool(machine_opts,

                                                  "kernel_irqchip", false);

        Error *err = NULL;



        if (irqchip_allowed) {

            icp = try_create_xics(TYPE_KVM_XICS, nr_servers, nr_irqs, &err);

        }

        if (irqchip_required && !icp) {

            error_report("kernel_irqchip requested but unavailable: %s",

                         error_get_pretty(err));

        }

    }



    if (!icp) {

        icp = try_create_xics(TYPE_XICS, nr_servers, nr_irqs, &error_abort);

    }



    return icp;

}
