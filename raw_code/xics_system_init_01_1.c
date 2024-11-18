static XICSState *xics_system_init(MachineState *machine,

                                   int nr_servers, int nr_irqs)

{

    XICSState *icp = NULL;



    if (kvm_enabled()) {

        Error *err = NULL;



        if (machine_kernel_irqchip_allowed(machine)) {

            icp = try_create_xics(TYPE_KVM_XICS, nr_servers, nr_irqs, &err);

        }

        if (machine_kernel_irqchip_required(machine) && !icp) {

            error_report("kernel_irqchip requested but unavailable: %s",

                         error_get_pretty(err));

        }


    }



    if (!icp) {

        icp = try_create_xics(TYPE_XICS, nr_servers, nr_irqs, &error_abort);

    }



    return icp;

}