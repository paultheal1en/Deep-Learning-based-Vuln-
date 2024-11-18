static void kvm_do_inject_x86_mce(void *_data)

{

    struct kvm_x86_mce_data *data = _data;

    int r;



    /* If there is an MCE exception being processed, ignore this SRAO MCE */

    if ((data->env->mcg_cap & MCG_SER_P) &&

        !(data->mce->status & MCI_STATUS_AR)) {

        r = kvm_mce_in_exception(data->env);

        if (r == -1) {

            fprintf(stderr, "Failed to get MCE status\n");

        } else if (r) {

            return;

        }

    }



    r = kvm_set_mce(data->env, data->mce);

    if (r < 0) {

        perror("kvm_set_mce FAILED");

        if (data->abort_on_error) {

            abort();

        }

    }

}
