static void qtest_process_command(CharDriverState *chr, gchar **words)

{

    const gchar *command;



    g_assert(words);



    command = words[0];



    if (qtest_log_fp) {

        qemu_timeval tv;

        int i;



        qtest_get_time(&tv);

        fprintf(qtest_log_fp, "[R +" FMT_timeval "]",

                (long) tv.tv_sec, (long) tv.tv_usec);

        for (i = 0; words[i]; i++) {

            fprintf(qtest_log_fp, " %s", words[i]);

        }

        fprintf(qtest_log_fp, "\n");

    }



    g_assert(command);

    if (strcmp(words[0], "irq_intercept_out") == 0

        || strcmp(words[0], "irq_intercept_in") == 0) {

        DeviceState *dev;

        NamedGPIOList *ngl;



        g_assert(words[1]);

        dev = DEVICE(object_resolve_path(words[1], NULL));

        if (!dev) {

            qtest_send_prefix(chr);

            qtest_send(chr, "FAIL Unknown device\n");

	    return;

        }



        if (irq_intercept_dev) {

            qtest_send_prefix(chr);

            if (irq_intercept_dev != dev) {

                qtest_send(chr, "FAIL IRQ intercept already enabled\n");

            } else {

                qtest_send(chr, "OK\n");

            }

	    return;

        }



        QLIST_FOREACH(ngl, &dev->gpios, node) {

            /* We don't support intercept of named GPIOs yet */

            if (ngl->name) {

                continue;

            }

            if (words[0][14] == 'o') {

                int i;

                for (i = 0; i < ngl->num_out; ++i) {

                    qemu_irq *disconnected = g_new0(qemu_irq, 1);

                    qemu_irq icpt = qemu_allocate_irq(qtest_irq_handler,

                                                      disconnected, i);



                    *disconnected = qdev_intercept_gpio_out(dev, icpt,

                                                            ngl->name, i);

                }

            } else {

                qemu_irq_intercept_in(ngl->in, qtest_irq_handler,

                                      ngl->num_in);

            }

        }

        irq_intercept_dev = dev;

        qtest_send_prefix(chr);

        qtest_send(chr, "OK\n");



    } else if (strcmp(words[0], "outb") == 0 ||

               strcmp(words[0], "outw") == 0 ||

               strcmp(words[0], "outl") == 0) {

        uint16_t addr;

        uint32_t value;



        g_assert(words[1] && words[2]);

        addr = strtoul(words[1], NULL, 0);

        value = strtoul(words[2], NULL, 0);



        if (words[0][3] == 'b') {

            cpu_outb(addr, value);

        } else if (words[0][3] == 'w') {

            cpu_outw(addr, value);

        } else if (words[0][3] == 'l') {

            cpu_outl(addr, value);

        }

        qtest_send_prefix(chr);

        qtest_send(chr, "OK\n");

    } else if (strcmp(words[0], "inb") == 0 ||

        strcmp(words[0], "inw") == 0 ||

        strcmp(words[0], "inl") == 0) {

        uint16_t addr;

        uint32_t value = -1U;



        g_assert(words[1]);

        addr = strtoul(words[1], NULL, 0);



        if (words[0][2] == 'b') {

            value = cpu_inb(addr);

        } else if (words[0][2] == 'w') {

            value = cpu_inw(addr);

        } else if (words[0][2] == 'l') {

            value = cpu_inl(addr);

        }

        qtest_send_prefix(chr);

        qtest_sendf(chr, "OK 0x%04x\n", value);

    } else if (strcmp(words[0], "writeb") == 0 ||

               strcmp(words[0], "writew") == 0 ||

               strcmp(words[0], "writel") == 0 ||

               strcmp(words[0], "writeq") == 0) {

        uint64_t addr;

        uint64_t value;



        g_assert(words[1] && words[2]);

        addr = strtoull(words[1], NULL, 0);

        value = strtoull(words[2], NULL, 0);



        if (words[0][5] == 'b') {

            uint8_t data = value;

            cpu_physical_memory_write(addr, &data, 1);

        } else if (words[0][5] == 'w') {

            uint16_t data = value;

            tswap16s(&data);

            cpu_physical_memory_write(addr, &data, 2);

        } else if (words[0][5] == 'l') {

            uint32_t data = value;

            tswap32s(&data);

            cpu_physical_memory_write(addr, &data, 4);

        } else if (words[0][5] == 'q') {

            uint64_t data = value;

            tswap64s(&data);

            cpu_physical_memory_write(addr, &data, 8);

        }

        qtest_send_prefix(chr);

        qtest_send(chr, "OK\n");

    } else if (strcmp(words[0], "readb") == 0 ||

               strcmp(words[0], "readw") == 0 ||

               strcmp(words[0], "readl") == 0 ||

               strcmp(words[0], "readq") == 0) {

        uint64_t addr;

        uint64_t value = UINT64_C(-1);



        g_assert(words[1]);

        addr = strtoull(words[1], NULL, 0);



        if (words[0][4] == 'b') {

            uint8_t data;

            cpu_physical_memory_read(addr, &data, 1);

            value = data;

        } else if (words[0][4] == 'w') {

            uint16_t data;

            cpu_physical_memory_read(addr, &data, 2);

            value = tswap16(data);

        } else if (words[0][4] == 'l') {

            uint32_t data;

            cpu_physical_memory_read(addr, &data, 4);

            value = tswap32(data);

        } else if (words[0][4] == 'q') {

            cpu_physical_memory_read(addr, &value, 8);

            tswap64s(&value);

        }

        qtest_send_prefix(chr);

        qtest_sendf(chr, "OK 0x%016" PRIx64 "\n", value);

    } else if (strcmp(words[0], "read") == 0) {

        uint64_t addr, len, i;

        uint8_t *data;

        char *enc;



        g_assert(words[1] && words[2]);

        addr = strtoull(words[1], NULL, 0);

        len = strtoull(words[2], NULL, 0);



        data = g_malloc(len);

        cpu_physical_memory_read(addr, data, len);



        enc = g_malloc(2 * len + 1);

        for (i = 0; i < len; i++) {

            sprintf(&enc[i * 2], "%02x", data[i]);

        }



        qtest_send_prefix(chr);

        qtest_sendf(chr, "OK 0x%s\n", enc);



        g_free(data);

        g_free(enc);

    } else if (strcmp(words[0], "b64read") == 0) {

        uint64_t addr, len;

        uint8_t *data;

        gchar *b64_data;



        g_assert(words[1] && words[2]);

        addr = strtoull(words[1], NULL, 0);

        len = strtoull(words[2], NULL, 0);



        data = g_malloc(len);

        cpu_physical_memory_read(addr, data, len);

        b64_data = g_base64_encode(data, len);

        qtest_send_prefix(chr);

        qtest_sendf(chr, "OK %s\n", b64_data);



        g_free(data);

        g_free(b64_data);

    } else if (strcmp(words[0], "write") == 0) {

        uint64_t addr, len, i;

        uint8_t *data;

        size_t data_len;



        g_assert(words[1] && words[2] && words[3]);

        addr = strtoull(words[1], NULL, 0);

        len = strtoull(words[2], NULL, 0);



        data_len = strlen(words[3]);

        if (data_len < 3) {

            qtest_send(chr, "ERR invalid argument size\n");

            return;

        }



        data = g_malloc(len);

        for (i = 0; i < len; i++) {

            if ((i * 2 + 4) <= data_len) {

                data[i] = hex2nib(words[3][i * 2 + 2]) << 4;

                data[i] |= hex2nib(words[3][i * 2 + 3]);

            } else {

                data[i] = 0;

            }

        }

        cpu_physical_memory_write(addr, data, len);

        g_free(data);



        qtest_send_prefix(chr);

        qtest_send(chr, "OK\n");

    } else if (strcmp(words[0], "memset") == 0) {

        uint64_t addr, len;

        uint8_t *data;

        uint8_t pattern;



        g_assert(words[1] && words[2] && words[3]);

        addr = strtoull(words[1], NULL, 0);

        len = strtoull(words[2], NULL, 0);

        pattern = strtoull(words[3], NULL, 0);



        if (len) {

            data = g_malloc(len);

            memset(data, pattern, len);

            cpu_physical_memory_write(addr, data, len);

            g_free(data);

        }



        qtest_send_prefix(chr);

        qtest_send(chr, "OK\n");

    }  else if (strcmp(words[0], "b64write") == 0) {

        uint64_t addr, len;

        uint8_t *data;

        size_t data_len;

        gsize out_len;



        g_assert(words[1] && words[2] && words[3]);

        addr = strtoull(words[1], NULL, 0);

        len = strtoull(words[2], NULL, 0);



        data_len = strlen(words[3]);

        if (data_len < 3) {

            qtest_send(chr, "ERR invalid argument size\n");

            return;

        }



        data = g_base64_decode_inplace(words[3], &out_len);

        if (out_len != len) {

            qtest_log_send("b64write: data length mismatch (told %"PRIu64", "

                           "found %zu)\n",

                           len, out_len);

            out_len = MIN(out_len, len);

        }



        cpu_physical_memory_write(addr, data, out_len);



        qtest_send_prefix(chr);

        qtest_send(chr, "OK\n");

    } else if (qtest_enabled() && strcmp(words[0], "clock_step") == 0) {

        int64_t ns;



        if (words[1]) {

            ns = strtoll(words[1], NULL, 0);

        } else {

            ns = qemu_clock_deadline_ns_all(QEMU_CLOCK_VIRTUAL);

        }

        qtest_clock_warp(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + ns);

        qtest_send_prefix(chr);

        qtest_sendf(chr, "OK %"PRIi64"\n",

                    (int64_t)qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL));

    } else if (qtest_enabled() && strcmp(words[0], "clock_set") == 0) {

        int64_t ns;



        g_assert(words[1]);

        ns = strtoll(words[1], NULL, 0);

        qtest_clock_warp(ns);

        qtest_send_prefix(chr);

        qtest_sendf(chr, "OK %"PRIi64"\n",

                    (int64_t)qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL));

    } else {

        qtest_send_prefix(chr);

        qtest_sendf(chr, "FAIL Unknown command '%s'\n", words[0]);

    }

}
