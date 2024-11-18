static void nvdimm_build_nfit(AcpiNVDIMMState *state, GArray *table_offsets,

                              GArray *table_data, BIOSLinker *linker)

{

    NvdimmFitBuffer *fit_buf = &state->fit_buf;

    unsigned int header;





    /* NVDIMM device is not plugged? */

    if (!fit_buf->fit->len) {

        return;

    }



    acpi_add_table(table_offsets, table_data);



    /* NFIT header. */

    header = table_data->len;

    acpi_data_push(table_data, sizeof(NvdimmNfitHeader));

    /* NVDIMM device structures. */

    g_array_append_vals(table_data, fit_buf->fit->data, fit_buf->fit->len);



    build_header(linker, table_data,

                 (void *)(table_data->data + header), "NFIT",

                 sizeof(NvdimmNfitHeader) + fit_buf->fit->len, 1, NULL, NULL);

}
