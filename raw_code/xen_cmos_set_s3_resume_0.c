void xen_cmos_set_s3_resume(void *opaque, int irq, int level)

{

    pc_cmos_set_s3_resume(opaque, irq, level);

    if (level) {

        xc_set_hvm_param(xen_xc, xen_domid, HVM_PARAM_ACPI_S_STATE, 3);

    }

}
