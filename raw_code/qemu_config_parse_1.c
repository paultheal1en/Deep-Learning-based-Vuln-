int qemu_config_parse(FILE *fp, QemuOptsList **lists, const char *fname)

{

    char line[1024], group[64], id[64], arg[64], value[1024];

    Location loc;

    QemuOptsList *list = NULL;

    QemuOpts *opts = NULL;

    int res = -1, lno = 0;



    loc_push_none(&loc);

    while (fgets(line, sizeof(line), fp) != NULL) {

        loc_set_file(fname, ++lno);

        if (line[0] == '\n') {

            /* skip empty lines */

            continue;

        }

        if (line[0] == '#') {

            /* comment */

            continue;

        }

        if (sscanf(line, "[%63s \"%63[^\"]\"]", group, id) == 2) {

            /* group with id */

            list = find_list(lists, group);

            if (list == NULL)

                goto out;

            opts = qemu_opts_create(list, id, 1);

            continue;

        }

        if (sscanf(line, "[%63[^]]]", group) == 1) {

            /* group without id */

            list = find_list(lists, group);

            if (list == NULL)

                goto out;

            opts = qemu_opts_create(list, NULL, 0);

            continue;

        }

        if (sscanf(line, " %63s = \"%1023[^\"]\"", arg, value) == 2) {

            /* arg = value */

            if (opts == NULL) {

                error_report("no group defined");

                goto out;

            }

            if (qemu_opt_set(opts, arg, value) != 0) {

                goto out;

            }

            continue;

        }

        error_report("parse error");

        goto out;

    }

    if (ferror(fp)) {

        error_report("error reading file");

        goto out;

    }

    res = 0;

out:

    loc_pop(&loc);

    return res;

}
