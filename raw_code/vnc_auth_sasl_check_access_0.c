static int vnc_auth_sasl_check_access(VncState *vs)

{

    const void *val;

    int err;



    err = sasl_getprop(vs->sasl.conn, SASL_USERNAME, &val);

    if (err != SASL_OK) {

	VNC_DEBUG("cannot query SASL username on connection %d (%s)\n",

		  err, sasl_errstring(err, NULL, NULL));

	return -1;

    }

    if (val == NULL) {

	VNC_DEBUG("no client username was found\n");

	return -1;

    }

    VNC_DEBUG("SASL client username %s\n", (const char *)val);



    vs->sasl.username = qemu_strdup((const char*)val);



    return 0;

}
