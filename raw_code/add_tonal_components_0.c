static int add_tonal_components(float *spectrum, int num_components,

                                TonalComponent *components)

{

    int i, j, last_pos = -1;

    float *input, *output;



    for (i = 0; i < num_components; i++) {

        last_pos = FFMAX(components[i].pos + components[i].num_coefs, last_pos);

        input    = components[i].coef;

        output   = &spectrum[components[i].pos];



        for (j = 0; j < components[i].num_coefs; j++)

            output[i] += input[i];

    }



    return last_pos;

}
