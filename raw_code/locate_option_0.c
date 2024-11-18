int locate_option(int argc, char **argv, const OptionDef *options,

                  const char *optname)

{

    const OptionDef *po;

    int i;



    for (i = 1; i < argc; i++) {

        const char *cur_opt = argv[i];



        if (*cur_opt++ != '-')

            continue;



        po = find_option(options, cur_opt);

        if (!po->name && cur_opt[0] == 'n' && cur_opt[1] == 'o')

            po = find_option(options, cur_opt + 2);



        if ((!po->name && !strcmp(cur_opt, optname)) ||

             (po->name && !strcmp(optname, po->name)))

            return i;



        if (!po || po->flags & HAS_ARG)

            i++;

    }

    return 0;

}
