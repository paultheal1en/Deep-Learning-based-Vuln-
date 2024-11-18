static void xen_read_physmap(XenIOState *state)

{

    XenPhysmap *physmap = NULL;

    unsigned int len, num, i;

    char path[80], *value = NULL;

    char **entries = NULL;



    snprintf(path, sizeof(path),

            "/local/domain/0/device-model/%d/physmap", xen_domid);

    entries = xs_directory(state->xenstore, 0, path, &num);

    if (entries == NULL)

        return;



    for (i = 0; i < num; i++) {

        physmap = g_malloc(sizeof (XenPhysmap));

        physmap->phys_offset = strtoull(entries[i], NULL, 16);

        snprintf(path, sizeof(path),

                "/local/domain/0/device-model/%d/physmap/%s/start_addr",

                xen_domid, entries[i]);

        value = xs_read(state->xenstore, 0, path, &len);

        if (value == NULL) {

            free(physmap);

            continue;

        }

        physmap->start_addr = strtoull(value, NULL, 16);

        free(value);



        snprintf(path, sizeof(path),

                "/local/domain/0/device-model/%d/physmap/%s/size",

                xen_domid, entries[i]);

        value = xs_read(state->xenstore, 0, path, &len);

        if (value == NULL) {

            free(physmap);

            continue;

        }

        physmap->size = strtoull(value, NULL, 16);

        free(value);



        snprintf(path, sizeof(path),

                "/local/domain/0/device-model/%d/physmap/%s/name",

                xen_domid, entries[i]);

        physmap->name = xs_read(state->xenstore, 0, path, &len);



        QLIST_INSERT_HEAD(&state->physmap, physmap, list);

    }

    free(entries);

}
