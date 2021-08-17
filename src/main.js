// let tutorial_data_path='https://storage.googleapis.com/tfjs-tutorials/carsData.json';
// async function getData() {
//     //拿到資料
//     const carsDataReq = await fetch(tutorial_data_path);
//     //轉成JSON
//     const carsData = await carsDataReq.json();
//     //將Miles_per_Gallon跟.Horsepower取出
//     //篩掉空資料
//     const cleaned = carsData.map(car => ({
//         mpg: car.Miles_per_Gallon,
//         horsepower: car.Horsepower,
//     })).filter(car => (car.mpg != null && car.horsepower != null));
//     //回傳清理過後的資料
//     return cleaned;
// }

// async function visualiztion() {
//     const values = (await getData()).map(d => ({
//         x: d.horsepower,
//         y: d.mpg,
//     }));
//     tfvis.render.scatterplot(
//         { name: 'Horsepower v MPG' },
//         { values }, 
//         {
//             xLabel: 'Horsepower',
//             yLabel: 'MPG',
//             height: 300
//         }
//     );
// }
// document.addEventListener('DOMContentLoaded', visualiztion);

// function createModel() {
//     // Create a sequential model
//     const model = tf.sequential();

//     // Add a single hidden layer
//     model.add(tf.layers.dense({ inputShape: [1], units: 20, useBias: true }));
//     model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));
//     model.add(tf.layers.dense({ units: 10, activation: 'sigmoid' }));
//     // Add an output layer
//     model.add(tf.layers.dense({ units: 1, useBias: true }));

//     // 加入最佳化的求解器、用MSE做為損失計算方式
//     model.compile({
//         optimizer: tf.train.adam(),
//         loss: tf.losses.meanSquaredError,
//         metrics: ['mse'],
//     });
//     return model;
// }

// function convertToTensor(data) {
//     // 使用tf.tidy讓除了回傳值以外，中間過程中的所佔用的空間釋放掉
//     return tf.tidy(() => {
//         // 打亂資料，在訓練最好都要做打亂資料的動作 
//         tf.util.shuffle(data);
//         // 將資料轉成tensor
//         const inputs = data.map(d => d.horsepower)
//         const labels = data.map(d => d.mpg);
//         const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
//         const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
//         //取最大值與最小值
//         const inputMax = inputTensor.max();
//         const inputMin = inputTensor.min();
//         const labelMax = labelTensor.max();
//         const labelMin = labelTensor.min();
//         //正規化 將 (tensor內的資料-最小值)/(最大值-最小值)) 出來的結果在0-1之間
//         const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
//         const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
//         return {
//             inputs: normalizedInputs,
//             labels: normalizedLabels,
//             inputMax,
//             inputMin,
//             labelMax,
//             labelMin,
//         };
//     });
// }

// async function trainModel(model, inputs, labels) {
//     //每次訓練的樣本數
//     const batchSize = 32;
//     //訓練多少代
//     const epochs = 50;
//     return await model.fit(inputs, labels, {
//         batchSize,
//         epochs,
//         shuffle: true,
//         callbacks: tfvis.show.fitCallbacks(
//             { name: 'Training Performance' },
//             ['loss', 'mse'],
//             { height: 200, callbacks: ['onEpochEnd'] }
//         )
//     });
// }

// function getPrediction(model, normalizationData) {
//     const { inputMax, inputMin, labelMin, labelMax } = normalizationData;
//     return tf.tidy(() => {
//         //tf.linspace(start_value,end_value,number_of_value);
//         const input_x = tf.linspace(0, 1, 100);
//         //將產生的資料轉成[num_examples, num_features_per_example]
//         const preds = model.predict(input_x.reshape([100, 1]));
//         //轉回原本的數= 數字*(最大值-最小值)+最小值
//         const toOrignalX = input_x
//             .mul(inputMax.sub(inputMin))
//             .add(inputMin);
//         const toOrignalY = preds
//             .mul(labelMax.sub(labelMin))
//             .add(labelMin);
//         //tensor.dataSync() return data from tensor to array
//         return [toOrignalX.dataSync(), toOrignalY.dataSync()];
//     });
// }

// function visualiztionPrediction(originalData,predictedData){
//     const originalPoints = originalData.map(d => ({
//         x: d.horsepower, y: d.mpg,
//     }));
//     const [px,py] = predictedData;
//     const predictedPoints = Array.from(px).map((val, i) => {
//         return { x: val, y: py[i] };
//     });
//     tfvis.render.scatterplot(
//     { name: 'Model Predictions vs Original Data' }, 
//     { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] }, 
//     {
//       xLabel: 'Horsepower',
//       yLabel: 'MPG',
//       height: 300
//     }
//   );
// }


// async function runTensorFlow(){
//     const model = createModel();
//     const data = await getData();
//     const tensorData = convertToTensor(data);
//     await trainModel(model, tensorData.inputs, tensorData.labels);
//     console.log('Done Training');
//     const predictedData = getPrediction(model,tensorData);
//     visualiztionPrediction(data,predictedData);
// }
// document.addEventListener('DOMContentLoaded', runTensorFlow);


fetch('./data/data.txt')
    .then(res => res.text())
    .then(txt => console.log(txt));







