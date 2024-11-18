void qmp_memsave(int64_t addr, int64_t size, const char *filename,

                 bool has_cpu, int64_t cpu_index, Error **errp)

{

    FILE *f;

    uint32_t l;

    CPUState *cpu;

    uint8_t buf[1024];



    if (!has_cpu) {

        cpu_index = 0;

    }



    cpu = qemu_get_cpu(cpu_index);

    if (cpu == NULL) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "cpu-index",

                  "a CPU number");

        return;

    }



    f = fopen(filename, "wb");

    if (!f) {

        error_setg_file_open(errp, errno, filename);

        return;

    }



    while (size != 0) {

        l = sizeof(buf);

        if (l > size)

            l = size;

        cpu_memory_rw_debug(cpu, addr, buf, l, 0);

        if (fwrite(buf, 1, l, f) != l) {

            error_set(errp, QERR_IO_ERROR);

            goto exit;

        }

        addr += l;

        size -= l;

    }



exit:

    fclose(f);

}
