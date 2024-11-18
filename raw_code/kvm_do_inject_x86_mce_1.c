static void kvm_do_inject_x86_mce(void *_data)

{

    struct kvm_x86_mce_data *data = _data;

    int r;



    r = kvm_set_mce(data->env, data->mce);

    if (r < 0)

        perror("kvm_set_mce FAILED");

}
