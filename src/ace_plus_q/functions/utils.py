

def scalar_complexmul(r1, im1, r2, im2):
    real_part = r1 * r2 - im1 * im2
    imag_part = im2 * r1 + im1 * r2

    return real_part, imag_part