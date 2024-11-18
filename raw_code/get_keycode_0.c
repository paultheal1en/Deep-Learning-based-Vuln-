static int get_keycode(const char *key)

{

    const KeyDef *p;

    char *endp;

    int ret;



    for(p = key_defs; p->name != NULL; p++) {

        if (!strcmp(key, p->name))

            return p->keycode;

    }

    if (strstart(key, "0x", NULL)) {

        ret = strtoul(key, &endp, 0);

        if (*endp == '\0' && ret >= 0x01 && ret <= 0xff)

            return ret;

    }

    return -1;

}
