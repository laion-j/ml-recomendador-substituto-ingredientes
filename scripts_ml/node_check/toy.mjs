import * as tf from "@tensorflow/tfjs-node";
const m = await tf.loadLayersModel("file://" + process.argv[2] + "/model.json");
m.predict(tf.zeros([2, 10])).print();
