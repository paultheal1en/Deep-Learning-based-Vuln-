static int spapr_fixup_cpu_dt(void *fdt, sPAPREnvironment *spapr)

{

    int ret = 0, offset, cpus_offset;

    CPUState *cs;

    char cpu_model[32];

    int smt = kvmppc_smt_threads();

    uint32_t pft_size_prop[] = {0, cpu_to_be32(spapr->htab_shift)};



    CPU_FOREACH(cs) {

        PowerPCCPU *cpu = POWERPC_CPU(cs);

        DeviceClass *dc = DEVICE_GET_CLASS(cs);

        int index = ppc_get_vcpu_dt_id(cpu);

        uint32_t associativity[] = {cpu_to_be32(0x5),

                                    cpu_to_be32(0x0),

                                    cpu_to_be32(0x0),

                                    cpu_to_be32(0x0),

                                    cpu_to_be32(cs->numa_node),

                                    cpu_to_be32(index)};



        if ((index % smt) != 0) {

            continue;

        }



        snprintf(cpu_model, 32, "%s@%x", dc->fw_name, index);



        cpus_offset = fdt_path_offset(fdt, "/cpus");

        if (cpus_offset < 0) {

            cpus_offset = fdt_add_subnode(fdt, fdt_path_offset(fdt, "/"),

                                          "cpus");

            if (cpus_offset < 0) {

                return cpus_offset;

            }

        }

        offset = fdt_subnode_offset(fdt, cpus_offset, cpu_model);

        if (offset < 0) {

            offset = fdt_add_subnode(fdt, cpus_offset, cpu_model);

            if (offset < 0) {

                return offset;

            }

        }



        if (nb_numa_nodes > 1) {

            ret = fdt_setprop(fdt, offset, "ibm,associativity", associativity,

                              sizeof(associativity));

            if (ret < 0) {

                return ret;

            }

        }



        ret = fdt_setprop(fdt, offset, "ibm,pft-size",

                          pft_size_prop, sizeof(pft_size_prop));

        if (ret < 0) {

            return ret;

        }



        ret = spapr_fixup_cpu_smt_dt(fdt, offset, cpu,

                                     smp_threads);

        if (ret < 0) {

            return ret;

        }

    }

    return ret;

}
