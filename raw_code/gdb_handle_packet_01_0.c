static int gdb_handle_packet(GDBState *s, const char *line_buf)

{

    CPUState *env;

    const char *p;

    int ch, reg_size, type, res, thread;

    char buf[MAX_PACKET_LENGTH];

    uint8_t mem_buf[MAX_PACKET_LENGTH];

    uint8_t *registers;

    target_ulong addr, len;



#ifdef DEBUG_GDB

    printf("command='%s'\n", line_buf);

#endif

    p = line_buf;

    ch = *p++;

    switch(ch) {

    case '?':

        /* TODO: Make this return the correct value for user-mode.  */

        snprintf(buf, sizeof(buf), "T%02xthread:%02x;", GDB_SIGNAL_TRAP,

                 s->c_cpu->cpu_index+1);

        put_packet(s, buf);

        /* Remove all the breakpoints when this query is issued,

         * because gdb is doing and initial connect and the state

         * should be cleaned up.

         */

        gdb_breakpoint_remove_all();

        break;

    case 'c':

        if (*p != '\0') {

            addr = strtoull(p, (char **)&p, 16);

#if defined(TARGET_I386)

            s->c_cpu->eip = addr;

#elif defined (TARGET_PPC)

            s->c_cpu->nip = addr;

#elif defined (TARGET_SPARC)

            s->c_cpu->pc = addr;

            s->c_cpu->npc = addr + 4;

#elif defined (TARGET_ARM)

            s->c_cpu->regs[15] = addr;

#elif defined (TARGET_SH4)

            s->c_cpu->pc = addr;

#elif defined (TARGET_MIPS)

            s->c_cpu->active_tc.PC = addr;

#elif defined (TARGET_CRIS)

            s->c_cpu->pc = addr;

#elif defined (TARGET_ALPHA)

            s->c_cpu->pc = addr;

#endif

        }

        s->signal = 0;

        gdb_continue(s);

	return RS_IDLE;

    case 'C':

        s->signal = gdb_signal_to_target (strtoul(p, (char **)&p, 16));

        if (s->signal == -1)

            s->signal = 0;

        gdb_continue(s);

        return RS_IDLE;

    case 'k':

        /* Kill the target */

        fprintf(stderr, "\nQEMU: Terminated via GDBstub\n");

        exit(0);

    case 'D':

        /* Detach packet */

        gdb_breakpoint_remove_all();

        gdb_continue(s);

        put_packet(s, "OK");

        break;

    case 's':

        if (*p != '\0') {

            addr = strtoull(p, (char **)&p, 16);

#if defined(TARGET_I386)

            s->c_cpu->eip = addr;

#elif defined (TARGET_PPC)

            s->c_cpu->nip = addr;

#elif defined (TARGET_SPARC)

            s->c_cpu->pc = addr;

            s->c_cpu->npc = addr + 4;

#elif defined (TARGET_ARM)

            s->c_cpu->regs[15] = addr;

#elif defined (TARGET_SH4)

            s->c_cpu->pc = addr;

#elif defined (TARGET_MIPS)

            s->c_cpu->active_tc.PC = addr;

#elif defined (TARGET_CRIS)

            s->c_cpu->pc = addr;

#elif defined (TARGET_ALPHA)

            s->c_cpu->pc = addr;

#endif

        }

        cpu_single_step(s->c_cpu, sstep_flags);

        gdb_continue(s);

	return RS_IDLE;

    case 'F':

        {

            target_ulong ret;

            target_ulong err;



            ret = strtoull(p, (char **)&p, 16);

            if (*p == ',') {

                p++;

                err = strtoull(p, (char **)&p, 16);

            } else {

                err = 0;

            }

            if (*p == ',')

                p++;

            type = *p;

            if (gdb_current_syscall_cb)

                gdb_current_syscall_cb(s->c_cpu, ret, err);

            if (type == 'C') {

                put_packet(s, "T02");

            } else {

                gdb_continue(s);

            }

        }

        break;

    case 'g':

        len = 0;

        for (addr = 0; addr < num_g_regs; addr++) {

            reg_size = gdb_read_register(s->g_cpu, mem_buf + len, addr);

            len += reg_size;

        }

        memtohex(buf, mem_buf, len);

        put_packet(s, buf);

        break;

    case 'G':

        registers = mem_buf;

        len = strlen(p) / 2;

        hextomem((uint8_t *)registers, p, len);

        for (addr = 0; addr < num_g_regs && len > 0; addr++) {

            reg_size = gdb_write_register(s->g_cpu, registers, addr);

            len -= reg_size;

            registers += reg_size;

        }

        put_packet(s, "OK");

        break;

    case 'm':

        addr = strtoull(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        len = strtoull(p, NULL, 16);

        if (cpu_memory_rw_debug(s->g_cpu, addr, mem_buf, len, 0) != 0) {

            put_packet (s, "E14");

        } else {

            memtohex(buf, mem_buf, len);

            put_packet(s, buf);

        }

        break;

    case 'M':

        addr = strtoull(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        len = strtoull(p, (char **)&p, 16);

        if (*p == ':')

            p++;

        hextomem(mem_buf, p, len);

        if (cpu_memory_rw_debug(s->g_cpu, addr, mem_buf, len, 1) != 0)

            put_packet(s, "E14");

        else

            put_packet(s, "OK");

        break;

    case 'p':

        /* Older gdb are really dumb, and don't use 'g' if 'p' is avaialable.

           This works, but can be very slow.  Anything new enough to

           understand XML also knows how to use this properly.  */

        if (!gdb_has_xml)

            goto unknown_command;

        addr = strtoull(p, (char **)&p, 16);

        reg_size = gdb_read_register(s->g_cpu, mem_buf, addr);

        if (reg_size) {

            memtohex(buf, mem_buf, reg_size);

            put_packet(s, buf);

        } else {

            put_packet(s, "E14");

        }

        break;

    case 'P':

        if (!gdb_has_xml)

            goto unknown_command;

        addr = strtoull(p, (char **)&p, 16);

        if (*p == '=')

            p++;

        reg_size = strlen(p) / 2;

        hextomem(mem_buf, p, reg_size);

        gdb_write_register(s->g_cpu, mem_buf, addr);

        put_packet(s, "OK");

        break;

    case 'Z':

    case 'z':

        type = strtoul(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        addr = strtoull(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        len = strtoull(p, (char **)&p, 16);

        if (ch == 'Z')

            res = gdb_breakpoint_insert(addr, len, type);

        else

            res = gdb_breakpoint_remove(addr, len, type);

        if (res >= 0)

             put_packet(s, "OK");

        else if (res == -ENOSYS)

            put_packet(s, "");

        else

            put_packet(s, "E22");

        break;

    case 'H':

        type = *p++;

        thread = strtoull(p, (char **)&p, 16);

        if (thread == -1 || thread == 0) {

            put_packet(s, "OK");

            break;

        }

        for (env = first_cpu; env != NULL; env = env->next_cpu)

            if (env->cpu_index + 1 == thread)

                break;

        if (env == NULL) {

            put_packet(s, "E22");

            break;

        }

        switch (type) {

        case 'c':

            s->c_cpu = env;

            put_packet(s, "OK");

            break;

        case 'g':

            s->g_cpu = env;

            put_packet(s, "OK");

            break;

        default:

             put_packet(s, "E22");

             break;

        }

        break;

    case 'T':

        thread = strtoull(p, (char **)&p, 16);

#ifndef CONFIG_USER_ONLY

        if (thread > 0 && thread < smp_cpus + 1)

#else

        if (thread == 1)

#endif

             put_packet(s, "OK");

        else

            put_packet(s, "E22");

        break;

    case 'q':

    case 'Q':

        /* parse any 'q' packets here */

        if (!strcmp(p,"qemu.sstepbits")) {

            /* Query Breakpoint bit definitions */

            snprintf(buf, sizeof(buf), "ENABLE=%x,NOIRQ=%x,NOTIMER=%x",

                     SSTEP_ENABLE,

                     SSTEP_NOIRQ,

                     SSTEP_NOTIMER);

            put_packet(s, buf);

            break;

        } else if (strncmp(p,"qemu.sstep",10) == 0) {

            /* Display or change the sstep_flags */

            p += 10;

            if (*p != '=') {

                /* Display current setting */

                snprintf(buf, sizeof(buf), "0x%x", sstep_flags);

                put_packet(s, buf);

                break;

            }

            p++;

            type = strtoul(p, (char **)&p, 16);

            sstep_flags = type;

            put_packet(s, "OK");

            break;

        } else if (strcmp(p,"C") == 0) {

            /* "Current thread" remains vague in the spec, so always return

             *  the first CPU (gdb returns the first thread). */

            put_packet(s, "QC1");

            break;

        } else if (strcmp(p,"fThreadInfo") == 0) {

            s->query_cpu = first_cpu;

            goto report_cpuinfo;

        } else if (strcmp(p,"sThreadInfo") == 0) {

        report_cpuinfo:

            if (s->query_cpu) {

                snprintf(buf, sizeof(buf), "m%x", s->query_cpu->cpu_index+1);

                put_packet(s, buf);

                s->query_cpu = s->query_cpu->next_cpu;

            } else

                put_packet(s, "l");

            break;

        } else if (strncmp(p,"ThreadExtraInfo,", 16) == 0) {

            thread = strtoull(p+16, (char **)&p, 16);

            for (env = first_cpu; env != NULL; env = env->next_cpu)

                if (env->cpu_index + 1 == thread) {

                    len = snprintf((char *)mem_buf, sizeof(mem_buf),

                                   "CPU#%d [%s]", env->cpu_index,

                                   env->halted ? "halted " : "running");

                    memtohex(buf, mem_buf, len);

                    put_packet(s, buf);

                    break;

                }

            break;

        }

#ifdef CONFIG_LINUX_USER

        else if (strncmp(p, "Offsets", 7) == 0) {

            TaskState *ts = s->c_cpu->opaque;



            snprintf(buf, sizeof(buf),

                     "Text=" TARGET_ABI_FMT_lx ";Data=" TARGET_ABI_FMT_lx

                     ";Bss=" TARGET_ABI_FMT_lx,

                     ts->info->code_offset,

                     ts->info->data_offset,

                     ts->info->data_offset);

            put_packet(s, buf);

            break;

        }

#endif

        if (strncmp(p, "Supported", 9) == 0) {

            snprintf(buf, sizeof(buf), "PacketSize=%x", MAX_PACKET_LENGTH);

#ifdef GDB_CORE_XML

            strcat(buf, ";qXfer:features:read+");

#endif

            put_packet(s, buf);

            break;

        }

#ifdef GDB_CORE_XML

        if (strncmp(p, "Xfer:features:read:", 19) == 0) {

            const char *xml;

            target_ulong total_len;



            gdb_has_xml = 1;

            p += 19;

            xml = get_feature_xml(p, &p);

            if (!xml) {

                snprintf(buf, sizeof(buf), "E00");

                put_packet(s, buf);

                break;

            }



            if (*p == ':')

                p++;

            addr = strtoul(p, (char **)&p, 16);

            if (*p == ',')

                p++;

            len = strtoul(p, (char **)&p, 16);



            total_len = strlen(xml);

            if (addr > total_len) {

                snprintf(buf, sizeof(buf), "E00");

                put_packet(s, buf);

                break;

            }

            if (len > (MAX_PACKET_LENGTH - 5) / 2)

                len = (MAX_PACKET_LENGTH - 5) / 2;

            if (len < total_len - addr) {

                buf[0] = 'm';

                len = memtox(buf + 1, xml + addr, len);

            } else {

                buf[0] = 'l';

                len = memtox(buf + 1, xml + addr, total_len - addr);

            }

            put_packet_binary(s, buf, len + 1);

            break;

        }

#endif

        /* Unrecognised 'q' command.  */

        goto unknown_command;



    default:

    unknown_command:

        /* put empty packet */

        buf[0] = '\0';

        put_packet(s, buf);

        break;

    }

    return RS_IDLE;

}
