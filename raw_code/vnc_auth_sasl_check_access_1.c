static int vnc_auth_sasl_check_access(VncState *vs)

{

    const void *val;

    int err;

    int allow;



    err = sasl_getprop(vs->sasl.conn, SASL_USERNAME, &val);

    if (err != SASL_OK) {

        VNC_DEBUG("cannot query SASL username on connection %d (%s), denying access\n",

                  err, sasl_errstring(err, NULL, NULL));

        return -1;

    }

    if (val == NULL) {

        VNC_DEBUG("no client username was found, denying access\n");

        return -1;

    }

    VNC_DEBUG("SASL client username %s\n", (const char *)val);



    vs->sasl.username = g_strdup((const char*)val);



    if (vs->vd->sasl.acl == NULL) {

        VNC_DEBUG("no ACL activated, allowing access\n");

        return 0;

    }



    allow = qemu_acl_party_is_allowed(vs->vd->sasl.acl, vs->sasl.username);



    VNC_DEBUG("SASL client %s %s by ACL\n", vs->sasl.username,

              allow ? "allowed" : "denied");

    return allow ? 0 : -1;

}
