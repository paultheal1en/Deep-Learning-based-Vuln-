static int spapr_post_load(void *opaque, int version_id)
{
    sPAPRMachineState *spapr = (sPAPRMachineState *)opaque;
    int err = 0;
    if (!object_dynamic_cast(OBJECT(spapr->ics), TYPE_ICS_KVM)) {
        CPUState *cs;
        CPU_FOREACH(cs) {
            PowerPCCPU *cpu = POWERPC_CPU(cs);
            icp_resend(ICP(cpu->intc));
    /* In earlier versions, there was no separate qdev for the PAPR
     * RTC, so the RTC offset was stored directly in sPAPREnvironment.
     * So when migrating from those versions, poke the incoming offset
     * value into the RTC device */
    if (version_id < 3) {
        err = spapr_rtc_import_offset(&spapr->rtc, spapr->rtc_offset);
    if (kvm_enabled() && spapr->patb_entry) {
        PowerPCCPU *cpu = POWERPC_CPU(first_cpu);
        bool radix = !!(spapr->patb_entry & PATBE1_GR);
        bool gtse = !!(cpu->env.spr[SPR_LPCR] & LPCR_GTSE);
        err = kvmppc_configure_v3_mmu(cpu, radix, gtse, spapr->patb_entry);
            error_report("Process table config unsupported by the host");
            return -EINVAL;