static int open_self_maps(void *cpu_env, int fd)

{

#if defined(TARGET_ARM) || defined(TARGET_M68K) || defined(TARGET_UNICORE32)

    CPUState *cpu = ENV_GET_CPU((CPUArchState *)cpu_env);

    TaskState *ts = cpu->opaque;

#endif

    FILE *fp;

    char *line = NULL;

    size_t len = 0;

    ssize_t read;



    fp = fopen("/proc/self/maps", "r");

    if (fp == NULL) {

        return -EACCES;

    }



    while ((read = getline(&line, &len, fp)) != -1) {

        int fields, dev_maj, dev_min, inode;

        uint64_t min, max, offset;

        char flag_r, flag_w, flag_x, flag_p;

        char path[512] = "";

        fields = sscanf(line, "%"PRIx64"-%"PRIx64" %c%c%c%c %"PRIx64" %x:%x %d"

                        " %512s", &min, &max, &flag_r, &flag_w, &flag_x,

                        &flag_p, &offset, &dev_maj, &dev_min, &inode, path);



        if ((fields < 10) || (fields > 11)) {

            continue;

        }

        if (!strncmp(path, "[stack]", 7)) {

            continue;

        }

        if (h2g_valid(min) && h2g_valid(max)) {

            dprintf(fd, TARGET_ABI_FMT_lx "-" TARGET_ABI_FMT_lx

                    " %c%c%c%c %08" PRIx64 " %02x:%02x %d %s%s\n",

                    h2g(min), h2g(max), flag_r, flag_w,

                    flag_x, flag_p, offset, dev_maj, dev_min, inode,

                    path[0] ? "         " : "", path);

        }

    }



    free(line);

    fclose(fp);



#if defined(TARGET_ARM) || defined(TARGET_M68K) || defined(TARGET_UNICORE32)

    dprintf(fd, "%08llx-%08llx rw-p %08llx 00:00 0          [stack]\n",

                (unsigned long long)ts->info->stack_limit,

                (unsigned long long)(ts->info->start_stack +

                                     (TARGET_PAGE_SIZE - 1)) & TARGET_PAGE_MASK,

                (unsigned long long)0);

#endif



    return 0;

}
