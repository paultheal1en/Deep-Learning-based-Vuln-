static int protocol_client_auth_sasl_mechname(VncState *vs, uint8_t *data, size_t len)

{

    char *mechname = malloc(len + 1);

    if (!mechname) {

        VNC_DEBUG("Out of memory reading mechname\n");

        vnc_client_error(vs);

    }

    strncpy(mechname, (char*)data, len);

    mechname[len] = '\0';

    VNC_DEBUG("Got client mechname '%s' check against '%s'\n",

              mechname, vs->sasl.mechlist);



    if (strncmp(vs->sasl.mechlist, mechname, len) == 0) {

        if (vs->sasl.mechlist[len] != '\0' &&

            vs->sasl.mechlist[len] != ',') {

            VNC_DEBUG("One %d", vs->sasl.mechlist[len]);

            vnc_client_error(vs);

            return -1;

        }

    } else {

        char *offset = strstr(vs->sasl.mechlist, mechname);

        VNC_DEBUG("Two %p\n", offset);

        if (!offset) {

            vnc_client_error(vs);

            return -1;

        }

        VNC_DEBUG("Two '%s'\n", offset);

        if (offset[-1] != ',' ||

            (offset[len] != '\0'&&

             offset[len] != ',')) {

            vnc_client_error(vs);

            return -1;

        }

    }



    free(vs->sasl.mechlist);

    vs->sasl.mechlist = mechname;



    VNC_DEBUG("Validated mechname '%s'\n", mechname);

    vnc_read_when(vs, protocol_client_auth_sasl_start_len, 4);

    return 0;

}
