void *bios_linker_loader_cleanup(GArray *linker)

{

    return g_array_free(linker, false);

}
