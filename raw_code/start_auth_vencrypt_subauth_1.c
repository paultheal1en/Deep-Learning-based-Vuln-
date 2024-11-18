static void start_auth_vencrypt_subauth(VncState *vs)

{

    switch (vs->subauth) {

    case VNC_AUTH_VENCRYPT_TLSNONE:

    case VNC_AUTH_VENCRYPT_X509NONE:

       VNC_DEBUG("Accept TLS auth none\n");

       vnc_write_u32(vs, 0); /* Accept auth completion */

       start_client_init(vs);

       break;



    case VNC_AUTH_VENCRYPT_TLSVNC:

    case VNC_AUTH_VENCRYPT_X509VNC:

       VNC_DEBUG("Start TLS auth VNC\n");

       start_auth_vnc(vs);

       break;



#ifdef CONFIG_VNC_SASL

    case VNC_AUTH_VENCRYPT_TLSSASL:

    case VNC_AUTH_VENCRYPT_X509SASL:

      VNC_DEBUG("Start TLS auth SASL\n");

      start_auth_sasl(vs);

      break;

#endif /* CONFIG_VNC_SASL */



    default: /* Should not be possible, but just in case */

       VNC_DEBUG("Reject subauth %d server bug\n", vs->auth);

       vnc_write_u8(vs, 1);

       if (vs->minor >= 8) {

           static const char err[] = "Unsupported authentication type";

           vnc_write_u32(vs, sizeof(err));

           vnc_write(vs, err, sizeof(err));

       }

       vnc_client_error(vs);

    }

}
