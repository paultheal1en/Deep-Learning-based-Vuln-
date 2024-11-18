void net_slirp_smb(const char *exported_dir)

{

    if (slirp_smb_export) {

        fprintf(stderr, "-smb given twice\n");

        exit(1);

    }

    slirp_smb_export = exported_dir;

    if (slirp_inited) {

        slirp_smb(exported_dir);

    }

}
