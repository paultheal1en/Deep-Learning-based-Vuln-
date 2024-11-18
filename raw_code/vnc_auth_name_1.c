static const char *vnc_auth_name(VncDisplay *vd) {

    switch (vd->auth) {

    case VNC_AUTH_INVALID:

        return "invalid";

    case VNC_AUTH_NONE:

        return "none";

    case VNC_AUTH_VNC:

        return "vnc";

    case VNC_AUTH_RA2:

        return "ra2";

    case VNC_AUTH_RA2NE:

        return "ra2ne";

    case VNC_AUTH_TIGHT:

        return "tight";

    case VNC_AUTH_ULTRA:

        return "ultra";

    case VNC_AUTH_TLS:

        return "tls";

    case VNC_AUTH_VENCRYPT:

#ifdef CONFIG_VNC_TLS

        switch (vd->subauth) {

        case VNC_AUTH_VENCRYPT_PLAIN:

            return "vencrypt+plain";

        case VNC_AUTH_VENCRYPT_TLSNONE:

            return "vencrypt+tls+none";

        case VNC_AUTH_VENCRYPT_TLSVNC:

            return "vencrypt+tls+vnc";

        case VNC_AUTH_VENCRYPT_TLSPLAIN:

            return "vencrypt+tls+plain";

        case VNC_AUTH_VENCRYPT_X509NONE:

            return "vencrypt+x509+none";

        case VNC_AUTH_VENCRYPT_X509VNC:

            return "vencrypt+x509+vnc";

        case VNC_AUTH_VENCRYPT_X509PLAIN:

            return "vencrypt+x509+plain";

        case VNC_AUTH_VENCRYPT_TLSSASL:

            return "vencrypt+tls+sasl";

        case VNC_AUTH_VENCRYPT_X509SASL:

            return "vencrypt+x509+sasl";

        default:

            return "vencrypt";

        }

#else

        return "vencrypt";

#endif

    case VNC_AUTH_SASL:

        return "sasl";

    }

    return "unknown";

}
