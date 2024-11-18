static void machine_set_loadparm(Object *obj, const char *val, Error **errp)

{

    S390CcwMachineState *ms = S390_CCW_MACHINE(obj);

    int i;



    for (i = 0; i < sizeof(ms->loadparm) && val[i]; i++) {

        uint8_t c = toupper(val[i]); /* mimic HMC */



        if (('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || (c == '.') ||

            (c == ' ')) {

            ms->loadparm[i] = c;

        } else {

            error_setg(errp, "LOADPARM: invalid character '%c' (ASCII 0x%02x)",

                       c, c);

            return;

        }

    }



    for (; i < sizeof(ms->loadparm); i++) {

        ms->loadparm[i] = ' '; /* pad right with spaces */

    }

}
