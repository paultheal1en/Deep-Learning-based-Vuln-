int acpi_table_add(const char *t)

{

    static const char *dfl_id = "QEMUQEMU";

    char buf[1024], *p, *f;

    struct acpi_table_header acpi_hdr;

    unsigned long val;

    uint32_t length;

    struct acpi_table_header *acpi_hdr_p;

    size_t off;



    memset(&acpi_hdr, 0, sizeof(acpi_hdr));

  

    if (get_param_value(buf, sizeof(buf), "sig", t)) {

        strncpy(acpi_hdr.signature, buf, 4);

    } else {

        strncpy(acpi_hdr.signature, dfl_id, 4);

    }

    if (get_param_value(buf, sizeof(buf), "rev", t)) {

        val = strtoul(buf, &p, 10);

        if (val > 255 || *p != '\0')

            goto out;

    } else {

        val = 1;

    }

    acpi_hdr.revision = (int8_t)val;



    if (get_param_value(buf, sizeof(buf), "oem_id", t)) {

        strncpy(acpi_hdr.oem_id, buf, 6);

    } else {

        strncpy(acpi_hdr.oem_id, dfl_id, 6);

    }



    if (get_param_value(buf, sizeof(buf), "oem_table_id", t)) {

        strncpy(acpi_hdr.oem_table_id, buf, 8);

    } else {

        strncpy(acpi_hdr.oem_table_id, dfl_id, 8);

    }



    if (get_param_value(buf, sizeof(buf), "oem_rev", t)) {

        val = strtol(buf, &p, 10);

        if(*p != '\0')

            goto out;

    } else {

        val = 1;

    }

    acpi_hdr.oem_revision = cpu_to_le32(val);



    if (get_param_value(buf, sizeof(buf), "asl_compiler_id", t)) {

        strncpy(acpi_hdr.asl_compiler_id, buf, 4);

    } else {

        strncpy(acpi_hdr.asl_compiler_id, dfl_id, 4);

    }



    if (get_param_value(buf, sizeof(buf), "asl_compiler_rev", t)) {

        val = strtol(buf, &p, 10);

        if(*p != '\0')

            goto out;

    } else {

        val = 1;

    }

    acpi_hdr.asl_compiler_revision = cpu_to_le32(val);

    

    if (!get_param_value(buf, sizeof(buf), "data", t)) {

         buf[0] = '\0';

    }



    length = sizeof(acpi_hdr);



    f = buf;

    while (buf[0]) {

        struct stat s;

        char *n = strchr(f, ':');

        if (n)

            *n = '\0';

        if(stat(f, &s) < 0) {

            fprintf(stderr, "Can't stat file '%s': %s\n", f, strerror(errno));

            goto out;

        }

        length += s.st_size;

        if (!n)

            break;

        *n = ':';

        f = n + 1;

    }



    if (!acpi_tables) {

        acpi_tables_len = sizeof(uint16_t);

        acpi_tables = qemu_mallocz(acpi_tables_len);

    }

    acpi_tables = qemu_realloc(acpi_tables,

                               acpi_tables_len + sizeof(uint16_t) + length);

    p = acpi_tables + acpi_tables_len;

    acpi_tables_len += sizeof(uint16_t) + length;



    *(uint16_t*)p = cpu_to_le32(length);

    p += sizeof(uint16_t);

    memcpy(p, &acpi_hdr, sizeof(acpi_hdr));

    off = sizeof(acpi_hdr);



    f = buf;

    while (buf[0]) {

        struct stat s;

        int fd;

        char *n = strchr(f, ':');

        if (n)

            *n = '\0';

        fd = open(f, O_RDONLY);



        if(fd < 0)

            goto out;

        if(fstat(fd, &s) < 0) {

            close(fd);

            goto out;

        }



        /* off < length is necessary because file size can be changed

           under our foot */

        while(s.st_size && off < length) {

            int r;

            r = read(fd, p + off, s.st_size);

            if (r > 0) {

                off += r;

                s.st_size -= r;

            } else if ((r < 0 && errno != EINTR) || r == 0) {

                close(fd);

                goto out;

            }

        }



        close(fd);

        if (!n)

            break;

        f = n + 1;

    }

    if (off < length) {

        /* don't pass random value in process to guest */

        memset(p + off, 0, length - off);

    }



    acpi_hdr_p = (struct acpi_table_header*)p;

    acpi_hdr_p->length = cpu_to_le32(length);

    acpi_hdr_p->checksum = acpi_checksum((uint8_t*)p, length);

    /* increase number of tables */

    (*(uint16_t*)acpi_tables) =

	    cpu_to_le32(le32_to_cpu(*(uint16_t*)acpi_tables) + 1);

    return 0;

out:

    if (acpi_tables) {

        qemu_free(acpi_tables);

        acpi_tables = NULL;

    }

    return -1;

}
