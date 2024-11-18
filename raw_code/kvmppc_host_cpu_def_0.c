const ppc_def_t *kvmppc_host_cpu_def(void)

{

    uint32_t host_pvr = mfpvr();

    const ppc_def_t *base_spec;

    ppc_def_t *spec;

    uint32_t vmx = kvmppc_get_vmx();

    uint32_t dfp = kvmppc_get_dfp();



    base_spec = ppc_find_by_pvr(host_pvr);



    spec = g_malloc0(sizeof(*spec));

    memcpy(spec, base_spec, sizeof(*spec));



    /* Now fix up the spec with information we can query from the host */



    alter_insns(&spec->insns_flags, PPC_ALTIVEC, vmx > 0);

    alter_insns(&spec->insns_flags2, PPC2_VSX, vmx > 1);

    alter_insns(&spec->insns_flags2, PPC2_DFP, dfp);



    return spec;

}
