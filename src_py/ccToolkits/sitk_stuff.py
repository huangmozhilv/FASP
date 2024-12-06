
import SimpleITK as sitk


# copied from nnunet
def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


def compare_geometry(image: sitk.Image, ref: sitk.Image, image_tag='lab', ref_tag='img'):
    res = dict()
    if image.GetOrigin()!=ref.GetOrigin():
        res['origins'] = 'not equal. {}:{}. {}:{}'.format(image_tag, str(image.GetOrigin()), ref_tag, str(ref.GetOrigin()))
    
    if image.GetDirection()!=ref.GetDirection():
        res['directions'] = 'not equal. {}:{}. {}:{}'.format(image_tag, str(image.GetDirection()), ref_tag, str(ref.GetDirection()))

    if image.GetSpacing()!=ref.GetSpacing():
        res['spacings'] = 'not equal. {}:{}. {}:{}'.format(image_tag, str(image.GetSpacing()), ref_tag, str(ref.GetSpacing()))
    
    return res