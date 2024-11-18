static void acpi_align_size(GArray *blob, unsigned align)

{

    /* Align size to multiple of given size. This reduces the chance

     * we need to change size in the future (breaking cross version migration).

     */

    g_array_set_size(blob, (ROUND_UP(acpi_data_len(blob), align) +

                            g_array_get_element_size(blob) - 1) /

                             g_array_get_element_size(blob));

}
