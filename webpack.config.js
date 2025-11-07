const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  // webpack: (config, { isServer }) => {
  //   if (!isServer) {
  //     // Копируем WASM файлы в public при клиентском билде
  //     config.plugins.push(
  //       new CopyPlugin({
  //         patterns: [
  //           {
  //             from: path.join(__dirname, 'node_modules/onnxruntime-web/dist/*.wasm'),
  //             to: path.join(__dirname, 'public/[name][ext]')
  //           }
  //         ]
  //       })
  //     );
  //   }

  //   return config;
  // },
  mode: "development",
  entry: "./src/index.ts",
  devtool: "inline-source-map",
  devServer: {
    static: "./public",
    port: 3000,
    open: true,
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: "ts-loader",
        exclude: /node_modules/
      },
      {
        test: /\.css$/,
        use: [
          "style-loader",
          "css-loader"
        ]
      }
    ]
  },
  resolve: {
    extensions: [
      ".ts",
      ".js"
    ]
  },
  output: {
    filename: "bundle.js",
    path: path.resolve(__dirname,
      "dist"),
    clean: true
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: "./public/index.html"
    })
  ],
  ignoreWarnings: [
    {
      module: /onnxruntime-web/,
      message: /Critical dependency/
    }
  ]
};
