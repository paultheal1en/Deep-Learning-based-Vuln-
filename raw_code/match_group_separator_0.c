static int match_group_separator(const OptionGroupDef *groups, const char *opt)

{

    const OptionGroupDef *p = groups;



    while (p->name) {

        if (p->sep && !strcmp(p->sep, opt))

            return p - groups;

        p++;

    }



    return -1;

}
