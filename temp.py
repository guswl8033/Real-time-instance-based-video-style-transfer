args = parser.parse_args()
################# DATA #################
def loadImg(imgPath):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(args.fineSize),
        transforms.ToTensor()])
    return transform(img)


style = loadImg(args.style).unsqueeze(0)

################# MODEL #################
if (args.layer == 'r31'):
    matrix = MulLayer(layer='r31')
    vgg = encoder3()
    dec = decoder3()
elif (args.layer == 'r41'):
    matrix = MulLayer(layer='r41')
    vgg = encoder4()
    dec = decoder4()
vgg.load_state_dict(torch.load(args.vgg_dir))
dec.load_state_dict(torch.load(args.decoder_dir))
matrix.load_state_dict(torch.load(args.matrixPath))
for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False

################# GLOBAL VARIABLE #################
content = torch.Tensor(1, 3, args.fineSize, args.fineSize)

################# GPU  #################
if (args.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()

    style = style.cuda()
    content = content.cuda()

totalTime = 0
imageCounter = 0
result_frames = []
contents = []
styles = []
cap = cv2.VideoCapture(0)  ##webcam
# videopath='/mnt/ITRC/LinearStyleTransfer/videos/' ##video
# cap = cv2.VideoCapture(videopath+'content1.avi')
cap.set(3, 256)
cap.set(4, 512)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(os.path.join(args.outf, args.name + '.avi'), fourcc, 20.0, (512, 256))
# out = cv2.VideoWriter(videopath+'content1_output1.avi',fourcc,20.0,(512,256))

with torch.no_grad():
    sF = vgg(style)

while (True):
    ret, frame = cap.read()
    if (ret == 0):
        break
    #    frame = cv2.resize(frame,dsize=(512,256),interpolation=cv2.INTER_CUBIC)
    cv2.imshow('origin', frame)
    frame = frame.transpose((2, 0, 1))
    frame = frame[::-1, :, :]
    frame = frame / 255.0
    frame = torch.from_numpy(frame.copy()).unsqueeze(0)
    # content.data.resize_(frame.size()).copy_(frame)
    with torch.no_grad():
        content.resize_(frame.size()).copy_(frame)
    with torch.no_grad():
        cF = vgg(content)
        if (args.layer == 'r41'):
            feature, transmatrix = matrix(cF[args.layer], sF[args.layer])
        else:
            feature, transmatrix = matrix(cF, sF)
        transfer = dec(feature)
    transfer = transfer.clamp(0, 1).squeeze(0).data.cpu().numpy()
    transfer = transfer.transpose((1, 2, 0))
    transfer = transfer[..., ::-1]
    out.write(np.uint8(transfer * 255))
    cv2.imshow('frame', transfer)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()