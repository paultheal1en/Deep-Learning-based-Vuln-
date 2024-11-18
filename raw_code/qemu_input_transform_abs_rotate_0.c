static void qemu_input_transform_abs_rotate(InputEvent *evt)

{

    InputMoveEvent *move = evt->u.abs;

    switch (graphic_rotate) {

    case 90:

        if (move->axis == INPUT_AXIS_X) {

            move->axis = INPUT_AXIS_Y;

        } else if (move->axis == INPUT_AXIS_Y) {

            move->axis = INPUT_AXIS_X;

            move->value = INPUT_EVENT_ABS_SIZE - 1 - move->value;

        }

        break;

    case 180:

        move->value = INPUT_EVENT_ABS_SIZE - 1 - move->value;

        break;

    case 270:

        if (move->axis == INPUT_AXIS_X) {

            move->axis = INPUT_AXIS_Y;

            move->value = INPUT_EVENT_ABS_SIZE - 1 - move->value;

        } else if (move->axis == INPUT_AXIS_Y) {

            move->axis = INPUT_AXIS_X;

        }

        break;

    }

}
