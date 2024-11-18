void nvdimm_init_acpi_state(AcpiNVDIMMState *state, MemoryRegion *io,

                            FWCfgState *fw_cfg, Object *owner)

{

    memory_region_init_io(&state->io_mr, owner, &nvdimm_dsm_ops, state,

                          "nvdimm-acpi-io", NVDIMM_ACPI_IO_LEN);

    memory_region_add_subregion(io, NVDIMM_ACPI_IO_BASE, &state->io_mr);



    state->dsm_mem = g_array_new(false, true /* clear */, 1);

    acpi_data_push(state->dsm_mem, TARGET_PAGE_SIZE);

    fw_cfg_add_file(fw_cfg, NVDIMM_DSM_MEM_FILE, state->dsm_mem->data,

                    state->dsm_mem->len);

}
