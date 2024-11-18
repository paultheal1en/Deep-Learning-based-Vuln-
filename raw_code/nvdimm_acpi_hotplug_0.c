void nvdimm_acpi_hotplug(AcpiNVDIMMState *state)

{

    nvdimm_build_fit_buffer(&state->fit_buf);

}
