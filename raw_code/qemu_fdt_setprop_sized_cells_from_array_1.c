int qemu_fdt_setprop_sized_cells_from_array(void *fdt,

                                            const char *node_path,

                                            const char *property,

                                            int numvalues,

                                            uint64_t *values)

{

    uint32_t *propcells;

    uint64_t value;

    int cellnum, vnum, ncells;

    uint32_t hival;



    propcells = g_new0(uint32_t, numvalues * 2);



    cellnum = 0;

    for (vnum = 0; vnum < numvalues; vnum++) {

        ncells = values[vnum * 2];

        if (ncells != 1 && ncells != 2) {

            return -1;

        }

        value = values[vnum * 2 + 1];

        hival = cpu_to_be32(value >> 32);

        if (ncells > 1) {

            propcells[cellnum++] = hival;

        } else if (hival != 0) {

            return -1;

        }

        propcells[cellnum++] = cpu_to_be32(value);

    }



    return qemu_fdt_setprop(fdt, node_path, property, propcells,

                            cellnum * sizeof(uint32_t));

}
