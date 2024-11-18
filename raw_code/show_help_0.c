void show_help(void)

{

    const OptionDef *po;

    int i, expert;



    printf("ffmpeg version " FFMPEG_VERSION ", Copyright (c) 2000,2001 Gerard Lantau\n"

           "usage: ffmpeg [[options] -i input_file]... {[options] outfile}...\n"

           "Hyper fast MPEG1/MPEG4/H263/RV and AC3/MPEG audio encoder\n"

           "\n"

           "Main options are:\n");

    for(i=0;i<2;i++) {

        if (i == 1)

            printf("\nAdvanced options are:\n");

        for(po = options; po->name != NULL; po++) {

            char buf[64];

            expert = (po->flags & OPT_EXPERT) != 0;

            if (expert == i) {

                strcpy(buf, po->name);

                if (po->flags & HAS_ARG) {

                    strcat(buf, " ");

                    strcat(buf, po->argname);

                }

                printf("-%-17s  %s\n", buf, po->help);

            }

        }

    }



    exit(1);

}
