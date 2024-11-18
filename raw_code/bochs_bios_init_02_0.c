static FWCfgState *bochs_bios_init(void)

{

    FWCfgState *fw_cfg;

    uint8_t *smbios_tables, *smbios_anchor;

    size_t smbios_tables_len, smbios_anchor_len;

    uint64_t *numa_fw_cfg;

    int i, j;

    unsigned int apic_id_limit = pc_apic_id_limit(max_cpus);



    fw_cfg = fw_cfg_init_io(BIOS_CFG_IOPORT);

    /* FW_CFG_MAX_CPUS is a bit confusing/problematic on x86:

     *

     * SeaBIOS needs FW_CFG_MAX_CPUS for CPU hotplug, but the CPU hotplug

     * QEMU<->SeaBIOS interface is not based on the "CPU index", but on the APIC

     * ID of hotplugged CPUs[1]. This means that FW_CFG_MAX_CPUS is not the

     * "maximum number of CPUs", but the "limit to the APIC ID values SeaBIOS

     * may see".

     *

     * So, this means we must not use max_cpus, here, but the maximum possible

     * APIC ID value, plus one.

     *

     * [1] The only kind of "CPU identifier" used between SeaBIOS and QEMU is

     *     the APIC ID, not the "CPU index"

     */

    fw_cfg_add_i16(fw_cfg, FW_CFG_MAX_CPUS, (uint16_t)apic_id_limit);

    fw_cfg_add_i32(fw_cfg, FW_CFG_ID, 1);

    fw_cfg_add_i64(fw_cfg, FW_CFG_RAM_SIZE, (uint64_t)ram_size);

    fw_cfg_add_bytes(fw_cfg, FW_CFG_ACPI_TABLES,

                     acpi_tables, acpi_tables_len);

    fw_cfg_add_i32(fw_cfg, FW_CFG_IRQ0_OVERRIDE, kvm_allows_irq0_override());



    smbios_tables = smbios_get_table_legacy(&smbios_tables_len);

    if (smbios_tables) {

        fw_cfg_add_bytes(fw_cfg, FW_CFG_SMBIOS_ENTRIES,

                         smbios_tables, smbios_tables_len);

    }



    smbios_get_tables(&smbios_tables, &smbios_tables_len,

                      &smbios_anchor, &smbios_anchor_len);

    if (smbios_anchor) {

        fw_cfg_add_file(fw_cfg, "etc/smbios/smbios-tables",

                        smbios_tables, smbios_tables_len);

        fw_cfg_add_file(fw_cfg, "etc/smbios/smbios-anchor",

                        smbios_anchor, smbios_anchor_len);

    }



    fw_cfg_add_bytes(fw_cfg, FW_CFG_E820_TABLE,

                     &e820_reserve, sizeof(e820_reserve));

    fw_cfg_add_file(fw_cfg, "etc/e820", e820_table,

                    sizeof(struct e820_entry) * e820_entries);



    fw_cfg_add_bytes(fw_cfg, FW_CFG_HPET, &hpet_cfg, sizeof(hpet_cfg));

    /* allocate memory for the NUMA channel: one (64bit) word for the number

     * of nodes, one word for each VCPU->node and one word for each node to

     * hold the amount of memory.

     */

    numa_fw_cfg = g_new0(uint64_t, 1 + apic_id_limit + nb_numa_nodes);

    numa_fw_cfg[0] = cpu_to_le64(nb_numa_nodes);

    for (i = 0; i < max_cpus; i++) {

        unsigned int apic_id = x86_cpu_apic_id_from_index(i);

        assert(apic_id < apic_id_limit);

        for (j = 0; j < nb_numa_nodes; j++) {

            if (test_bit(i, numa_info[j].node_cpu)) {

                numa_fw_cfg[apic_id + 1] = cpu_to_le64(j);

                break;

            }

        }

    }

    for (i = 0; i < nb_numa_nodes; i++) {

        numa_fw_cfg[apic_id_limit + 1 + i] = cpu_to_le64(numa_info[i].node_mem);

    }

    fw_cfg_add_bytes(fw_cfg, FW_CFG_NUMA, numa_fw_cfg,

                     (1 + apic_id_limit + nb_numa_nodes) *

                     sizeof(*numa_fw_cfg));



    return fw_cfg;

}
